# Octree sharding codec

Defines an `array -> bytes` codec that shards an array like
[`sharding_indexed`](../sharding_indexed/): each shard is one stored object
holding many inner chunks.  The array, its chunk grid (the shards), the inner
chunk shape and codecs, the chunk key encoding and the fill value mean exactly
what they mean with `sharding_indexed`.  Only the bytes of a shard differ:

- the inner chunks are stored in Z-order (Morton order), contiguously;
- they are grouped into **runs**, consecutive chunks a writer intends to be
  read together, such as the leaves of an octree aimed at a target size;
- the index is **sparse**: one row per stored inner chunk, not per possible
  one, plus a table of the runs.

The codec is meant for sparse, spatially clustered data such as point clouds,
meshes and skeletons binned into a grid, where most inner chunks of a shard
are empty and occupied ones vary widely in size.  It is named for its main use,
three-dimensional data, but is defined for any number of dimensions (in two it
is a quadtree).

## Motivation

*This section is non-normative.*

`sharding_indexed` stores a dense index of 16 bytes per *possible* inner
chunk.  For a sparse array that index can be larger than the data: a shard of
16³ inner chunks carries a 64 KiB index however few chunks it holds.  The
`octree_sharding` index costs 6 bytes per *stored* chunk (for shards of up to
32³ inner chunks) and nothing for absent ones.

The run table records a grouping of the chunks that the writer chose, such as
octree leaves of about equal weight.  Consumers that work in bounded units
(GPU batches, graph partitions) can take whole runs, and a reader can fetch a
run's wanted chunks with one byte range.  A `sharding_indexed` writer may also
store its chunks in Z-order, but its index cannot record the grouping.

On a clustered point cloud of 2 M points in a 128³ grid of cells, with one
inner chunk per cell (the points' float32 coordinates, zstd), sharded 16³
cells per shard, and read through zarr-python 3.4.2 behind a store with 20 ms
latency and 100 MB/s per request:

| | `sharding_indexed` | `sharding_indexed`, chunks in Z-order | `octree_sharding` |
| - | - | - | - |
| store size | 64.0 MB | 64.0 MB | 33.1 MB |
| of which index | 33.6 MB (64 KiB a shard) | 33.6 MB | 2.7 MB (5.1 KiB a shard) |
| 8³-cell box: requests, data read, time | 4, 0.66 MB, 121 ms | 4, 0.59 MB, 121 ms | 8, 0.36 MB, 113 ms |
| 32³-cell box: requests, data read, time | 53, 5.7 MB, 5.0 s | 53, 5.5 MB, 6.0 s | 78, 4.4 MB, 4.7 s |

(Medians over 10 boxes centred on random points.  The raw coordinates are
24 MB.)

The sparse index halves this store and cuts the data a query reads by a
quarter.  Query times are similar: zarr-python 3.4 already batches nearby
ranges of a `sharding_indexed` shard into few requests, and its reader of
this codec makes one request per run touched.  Placing `sharding_indexed`'s
chunks in Z-order changes nothing through zarr.  So the case for the codec is
its index size, and runs as units for consumers that want them, not query
speed.  The benchmark is `bench/bench_registry.py` in the implementation's
repository.

## Alternatives considered

*This section is non-normative.*

**Compressing `sharding_indexed`'s index.**  A sparse shard's dense index is
mostly the `0xFFFFFFFFFFFFFFFF` marker of absent chunks and would compress to
almost nothing.  The `sharding_indexed` specification rules this out: codecs
"that produce variable-sized encoded representation, such as compression
codecs, MUST NOT be used for index codecs", because a reader locates the index
by its fixed size.  A sparse index needs a variable size, and so a footer that
records it, which `sharding_indexed` has no place for.

**Storing `sharding_indexed`'s chunks in Z-order.**  Allowed already (the
specification leaves the order to the writer), but the index stays dense and
cannot record runs.  The benchmark above shows it changes nothing when read
through zarr-python.

**A new index format for `sharding_indexed`.**  This codec could become a
sparse-index option of a future version of the sharding specification.  It is
registered as its own codec so that it can be used now; its metadata is that
of `sharding_indexed` less the two index fields, so folding it in later would
be straightforward.

**Compared with `sharding_indexed`:**

| | `sharding_indexed` | `octree_sharding` |
| - | - | - |
| inner chunks, inner codecs, chunk keys, fill value | | the same |
| index rows | one per possible inner chunk (absent ones marked) | one per stored inner chunk |
| index row | offset and length, 16 bytes | Z-order code and length, 6 bytes for shards of up to 32³ inner chunks |
| index size | fixed: 16 bytes × possible chunks, + 4 | `36 + (c + w)·m + 12·r` bytes, with the footer |
| chunk order | any; gaps allowed | increasing Z-order, contiguous, no gaps (so offsets are implied) |
| locating the index | fixed size, at the start or end | a fixed 16-byte footer gives its size |
| index encoding | `index_codecs`, fixed-size only | fixed by this specification, CRC-32C checksum |
| grouping of chunks | none | runs, with a weight each |
| replacing one inner chunk | may rewrite it in place | rewrites the shard |

**Sparse volumetric formats inside a chunk** (for example a
[NanoVDB codec](https://github.com/zarr-developers/zarr-extensions/pull/73))
encode a sparse block's *values* in one chunk with a tree of their own.  This
codec changes only how a shard finds its inner chunks; what an inner chunk
holds, and how it is encoded, is left to the inner codecs.  The two can be
combined.

## Codec metadata

| Field | Type | Required |
| - | - | - |
| `name` | string | yes |
| `configuration` | object | yes |

### Name

The value of the `name` field MUST be the string `"octree_sharding"`.

### Configuration

| Field | Type | Required |
| - | - | - |
| [`chunk_shape`](#chunk_shape) | array of positive integers | yes |
| [`codecs`](#codecs) | array of codec metadata | yes |
| [`layout`](#layout) | object | no |

Additional keys are reserved for future versions of this codec.  Metadata with
additional keys MUST be treated as invalid by readers.  In particular,
`sharding_indexed`'s `index_codecs` and `index_location` are not part of this
codec: its index format and location are fixed.

#### chunk_shape

The shape of the inner chunks, as for `sharding_indexed`.  The shard shape
(the array's chunk grid) MUST be divisible by `chunk_shape` along every
dimension.

#### codecs

The codecs each inner chunk is encoded with, as for `sharding_indexed`.

#### layout

How a writer groups inner chunks into runs.  It only steers writers; decoding
never depends on it.  Readers MUST NOT reject an array because they do not
support its `layout` or its `unit`.  Writers SHOULD follow it when they
support it, and MAY group runs by another rule when they do not.

The value is an object `{"name": ..., "configuration": {...}}`:

- `{"name": "octree", "configuration": {"target": T, "unit": U}}`: the runs are
  the leaves of an octree aimed at weight `T` (see [Writing](#writing)).
  `target` is a positive number; the default is `100000`.
- `{"name": "runs", "configuration": {"min_weight": A, "max_weight": B, "unit": U}}`:
  the runs are Z-order runs with weights between `A` and `B`, `0 <= A <= B`,
  `B > 0`.

`unit` names what a weight counts.  `"bytes"`, the default, is the encoded
length of an inner chunk.  Other values are defined by implementations (for
example, the number of points a chunk holds).  Without `layout`, a writer MAY
group runs as it likes.

## Terms

- **chunks per shard** `s = (s_0, …, s_{n-1})`: the shard shape divided by
  `chunk_shape`, where `n` is the number of dimensions.
- **bits** `b = max(1, bit_length(max(s) − 1))`: a shard is addressed as a
  block of `2^b` inner chunks along every axis.  `n · b` MUST NOT exceed 64,
  and `n` MUST NOT exceed 255.  Chunks per shard need not be powers of two or
  equal along every axis: the block then extends past the shard, and codes of
  positions outside it never occur.  For example, `s = (3, 3, 3)` gives
  `b = 2`, and a code whose coordinate is 3 on any axis is invalid.
- **code**: an inner chunk's Morton (Z-order) code within its shard.  Bit `j`
  of the chunk's coordinate along axis `a` (counting from 0, the least
  significant bit) is bit `j · n + (n − 1 − a)` of the code, so axis 0 is the
  most significant.  Every aligned octree block of the shard is a contiguous
  range of codes.
- **run**: one or more stored inner chunks, consecutive in code order, read as
  one byte range.
- **weight**: a number a writer records for each run (for example, its size
  in bytes).  Readers ignore it.

## Shard format

All integers and floats are little endian.  A shard object is:

```
[inner chunks][index][footer]
```

**Inner chunks.**  Every stored inner chunk's encoded bytes, in increasing code
order, contiguous from byte 0, with no bytes between chunks and none between
the last chunk and the index.  Offsets are not stored: chunk `i` starts at the
sum of the lengths of the chunks before it.  At most `2³² − 1` chunks are
stored in a shard.

**Index.**

| Field | Type |
| - | - |
| magic | 4 bytes, `ZOSI` |
| version | uint8, `3` |
| `n` (dimensions) | uint8 |
| `b` (bits) | uint8 |
| length width `w` | uint8, `4` or `8` |
| stored chunk count `m` | uint32 |
| run count `r` | uint32 |
| codes | `m` × uint16, uint32 or uint64: the narrowest that holds `n · b` bits; strictly increasing |
| lengths | `m` × uint32 (`w = 4`) or uint64 (`w = 8`): each chunk's encoded length |
| run starts | `(r + 1)` × uint32: the index of each run's first chunk; the last entry is `m` |
| run weights | `r` × float64 |

A writer SHOULD use `w = 4` unless a chunk's encoded length is at least 2³².

**Footer.**  16 bytes.

| Field | Type |
| - | - |
| index size in bytes | uint64 |
| checksum | uint32: CRC-32C (Castagnoli, as in the [`crc32c`](../crc32c/) codec) of the index bytes |
| magic | 4 bytes, `ZOSF` |

An inner chunk with no row is absent, and reads as the fill value.  A shard
with no stored chunk SHOULD have no object (and then reads as the fill value,
as for any missing shard).  A reader MUST accept a shard object with `m = 0`
and `r = 0` (an index and footer only) and read it as the fill value.

Run weights carry no meaning for reading: a reader MUST NOT reject a shard for
its weights' values (including NaN or infinity), and MAY ignore them.

**Versions.**  This document defines index version 3.  A reader MUST reject an
index whose version it does not support.  A change to the index layout takes a
new version number and an entry in the [change log](#change-log).

## Reading

1. Read the end of the shard.  A reader MAY read any length, then read again
   if the footer's index size says the index is longer.  The index and footer
   of a shard with `m` stored chunks in `r` runs take `36 + (c + w)·m + 12·r`
   bytes, where `c` is the width of a code.  So for shards of up to 32³ inner
   chunks (`c = 2`), a suffix of `36 + 9·P` bytes, where `P` is the number of
   possible inner chunks, holds the whole index in one read whenever
   `r <= P/4`.
2. Check the footer magic and the checksum, then decode the index.  A reader
   MUST treat the shard as invalid, and not return its data, unless:
   - the index magic is `ZOSI` and the version is one the reader supports;
   - `n` is the array's dimensionality and `b` follows from its chunks per
     shard;
   - `w` is 4 or 8, and the index size equals the size its counts imply;
   - the codes are strictly increasing, and every code decodes to a
     coordinate inside the shard (below `s` on every axis);
   - every length is positive;
   - `r = 0` exactly when `m = 0`, the run starts begin at 0, end at `m` and
     increase strictly;
   - when the whole shard is read, the inner chunks' lengths sum to the size
     of the data before the index.
3. For each wanted inner chunk, find its code among the codes (by binary
   search), then fetch its bytes and decode them with the inner codecs.  A
   reader MAY fetch each chunk alone, or one range per run spanning the run's
   wanted chunks, or any other ranges.  A range that extends past the end of
   the shard means the shard is invalid.

## Writing

1. Encode each inner chunk with the inner codecs.  Inner chunks that hold only
   the fill value MAY be left out, as with `sharding_indexed`.
2. Order the stored chunks by code and lay their bytes out contiguously.
3. Partition them into runs: any partition into ranges of consecutive stored
   chunks is valid.  `layout` names the rule the writer intends.
4. Append the index (version 3) and the footer.  If no chunk is stored, delete
   the shard object instead.

Changing part of a shard rewrites the whole shard, as with
`sharding_indexed`.

### The `octree` layout

*Non-normative: a writer may group runs by any rule.*

Start from the shard's root block (`b` levels).  A block heavier than `target`,
holding more than one stored chunk, is split into its `2^n` children when its
occupied children are on average closer to `target` in log weight than the
block itself.  Otherwise the block is a leaf.  Every leaf is an aligned octree
block, so it is a range of consecutive codes: the leaves in code order are the
runs.  No weight is a hard bound, so leaf weights spread on both sides of
`target`.

### The `runs` layout

*Non-normative.*

Walk the stored chunks in code order.  Each run takes chunks until adding the
next would exceed `max_weight`.  It ends at the candidate boundary with weight
in `[min_weight, max_weight]` that lies on the coarsest octree block boundary,
and avoids leaving a last run lighter than `min_weight` where it can.  A single
chunk heavier than `max_weight` is a run of its own.

## Example

An array of 4³ `uint8` in one shard of 2³ inner chunks of 2³, with three inner
chunks stored and runs aimed at 8 bytes:

```json
{
  "name": "octree_sharding",
  "configuration": {
    "chunk_shape": [2, 2, 2],
    "codecs": [{ "name": "bytes" }],
    "layout": { "name": "octree", "configuration": { "target": 8.0, "unit": "bytes" } }
  }
}
```

The shard `c/0/0/0`, 114 bytes:

```
  0  01 01 01 01 01 01 01 01   inner chunk (0, 0, 0), code 0
  8  02 02 02 02 02 02 02 02   inner chunk (0, 1, 0), code 2
 16  03 03 03 03 03 03 03 03   inner chunk (1, 1, 1), code 7
 24  5a 4f 53 49   index magic "ZOSI"
 28  03   version
 29  03   n
 30  01   b
 31  04   length width
 32  03 00 00 00   stored chunk count
 36  03 00 00 00   run count
 40  00 00 02 00 07 00   codes (uint16)
 46  08 00 00 00 08 00 00 00 08 00 00 00   lengths (uint32)
 58  00 00 00 00 01 00 00 00 02 00 00 00 03 00 00 00   run starts (uint32)
 74  00 00 00 00 00 00 20 40 (×3)   run weights (float64, 8.0 each)
 98  4a 00 00 00 00 00 00 00   footer: index size (74)
106  6e 2f 2c 71   footer: CRC-32C
110  5a 4f 53 46   footer: magic "ZOSF"
```

The array is in the implementation's repository as
[`registry/codecs/octree_sharding/example.zarr`](https://github.com/Andrew-Keenlyside/zarr-octree/tree/main/registry/codecs/octree_sharding/example.zarr),
written by `registry/make_example.py`.

## Interoperability

- An implementation without this codec refuses the array (an unknown codec),
  so it can never misread a shard.
- The array metadata is otherwise that of a `sharding_indexed` array, so
  converting between the two codecs rewrites only the shard objects and the
  codec entry.
- The codec places no limit on the data type or the inner codecs.

## Legacy name `zarr_octree_sharding`

Before registration, the implementation wrote this format under the name
`zarr_octree_sharding`, with a flat configuration (`version`: `2`, `layout` as
a string, and the layout's fields beside it) and index version 2.  Version 2
differs from version 3 only in its checksum: CRC-32 (ISO-HDLC, as in zlib)
instead of CRC-32C.  Implementations MAY read that name and index version 2,
and MUST NOT write either.

## Implementations

- Python: [zarr-octree](https://github.com/Andrew-Keenlyside/zarr-octree), a
  codec for zarr-python 3.2 and later, registered through its `zarr.codecs`
  entry point.

## References

- [`sharding_indexed`](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/)
- Morton, G. M. *A computer oriented geodetic data base and a new technique in
  file sequencing.* IBM, 1966.

## Change log

No changes yet.

## Current maintainers

- Andrew Keenlyside ([@Andrew-Keenlyside](https://github.com/Andrew-Keenlyside))
