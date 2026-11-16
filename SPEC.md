# `zarr_octree_sharding` — shard format (draft, index version 2)

A Zarr v3 array-to-bytes codec in place of `sharding_indexed`. The array's
data model is unchanged: shape, dtype, the chunk grid (the shards), the inner
chunk shape and codecs, the chunk key encoding and the fill value mean exactly
what they mean with `sharding_indexed`. Only the bytes of a shard object
differ: the inner chunks are stored in Z-order, grouped into **runs**, with a
**sparse index** at the end.

## Metadata

```json
{
  "name": "zarr_octree_sharding",
  "configuration": {
    "version": 2,
    "chunk_shape": [16, 16, 16],
    "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "zstd"}],
    "layout": "octree",
    "target": 100000,
    "unit": "bytes"
  }
}
```

* `chunk_shape`, `codecs`: as for `sharding_indexed` (inner chunk shape and
  the codecs each inner chunk is encoded with).
* `layout`: `"octree"` with `target`, or `"runs"` with `min_weight` and
  `max_weight`. Metadata without `layout` was written before the octree layout
  and means `"runs"`.
* `unit`: what weights count — `"bytes"` (encoded inner-chunk bytes) or a name
  the writer registered a weigher under (e.g. vertices).

`layout`, `target`, the bounds and `unit` steer how writers cut runs; readers
ignore them. A reader that does not know the codec refuses the array (unknown
codec), so it can never misread a shard.

## Terms

* **chunks per shard** `s = (s_0, …, s_{n-1})`: the shard shape divided by the
  inner chunk shape.
* **bits** `b = max(1, bit_length(max(s) − 1))`: a shard is addressed as a
  block of `2^b` inner chunks per axis.
* **code**: an inner chunk's Morton (Z-order) code within the shard — the bits
  of its coordinates interleaved, bit `j` of axis `a` at position
  `j·n + (n − 1 − a)` (axis 0 most significant). Every aligned octree block of
  the shard is one contiguous range of codes.
* **run**: consecutive stored inner chunks (in code order) read as one byte
  range. With `layout="octree"` the runs are the leaves of an octree: a block
  heavier than `target` is split into its `2^n` children when the occupied
  children are on average closer to `target` in log weight than the block;
  no weight is a hard bound. With `layout="runs"` they are Z-order runs within
  `[min_weight, max_weight]`, cut on the coarsest octree boundary available
  (a single inner chunk may exceed `max_weight`).

## Shard object (little endian)

```
[inner chunks, in code order, contiguous from byte 0][index][footer]
```

**Index**

| field | type |
|-------|------|
| magic `ZOSI` | 4 bytes |
| version (2) | uint8 |
| ndim `n` | uint8 |
| bits `b` | uint8 |
| length width `w` (4 or 8) | uint8 |
| stored chunk count `m` | uint32 |
| run count `r` | uint32 |
| codes, sorted | `m` × uint16 / uint32 / uint64 — the narrowest holding `n·b` bits |
| encoded lengths | `m` × uint`8w` (`w = 8` only when a chunk is ≥ 4 GiB) |
| first chunk of each run | `(r + 1)` × uint32 (the last entry is `m`) |
| weight of each run | `r` × float64 |

Offsets are not stored: chunk `i` starts at the sum of the lengths before it.

**Footer**

| field | type |
|-------|------|
| index size in bytes | uint64 |
| CRC-32 of the index | uint32 |
| magic `ZOSF` | 4 bytes |

An inner chunk not listed is absent (fill value). A shard with no stored
chunk has no object (writing one empty deletes it).

## Reading

1. Fetch the end of the shard with one suffix request. A reader may take any
   length; the reference reader takes room for every possible chunk's row plus
   a run table of a quarter as many runs (at least 4 KiB), and reads again only
   when the footer's index size says the index is longer.
2. Check the footer magic and the index CRC; decode the index.
3. For each wanted inner chunk, find its code among the sorted codes. Fetch one
   byte range per run that holds a wanted chunk (or merge ranges as the reader
   likes), then decode each chunk with the inner codecs.

At 2 + 4 bytes per stored chunk (shards up to 32³ chunks), the sparse index is
about a third of `sharding_indexed`'s dense one (16 bytes per *possible*
chunk) for a full shard, and much less for a sparse one.

## Writing

A writer stores the shard's present inner chunks in code order, partitions
them into runs by its layout, and writes the whole shard. A partial write
reads the shard, replaces the changed chunks and re-cuts the runs. How runs are
chosen is not part of the format; readers only rely on the index.
