# `zarr_octree` — octree chunk packing for Zarr v3 (draft, format version 2)

A storage-layer extension.  The array's data model is unchanged: shape,
dtype, chunk grid (regular), chunk key encoding, codecs and fill value mean
exactly what they mean without the extension.  Only *where the encoded chunk
bytes live* changes: instead of one storage object per chunk, the chunks are
grouped into **nodes**, each holding the encoded bytes of every occupied
chunk in a run of the chunk grid's octree Z-order.  Nodes are sized by
content — between a minimum and a maximum weight, in bytes or in a unit the
writer measures such as vertices — so a dense region is cut into runs of few
chunks and a sparse region into runs of many.

Format version 1 stored whole octree nodes, which bounds a node's size from
above only: a sparse block next to a dense sibling stays small.  Version 2
merges siblings along the Z-order, which bounds it from both sides.

## Terms

* **cell** — one chunk of the array, addressed by its chunk coordinates
  `(i, j, k, …)`.  The encoded bytes of a cell are exactly what a regular
  store would hold at the chunk key.
* **max depth `D`** — a **root** covers `2^D` cells along every axis.  Roots
  are partitioned independently.  `n · D ≤ 64` for an `n`-d array.
* **code** — a cell's Morton (Z-order) code within its root: the bits of its
  local coordinates `x − (x >> D << D)` interleaved, bit `b` of axis `a` at
  position `b·n + (n − 1 − a)`.  Every aligned octree block of a root is one
  contiguous range of codes.
* **node** — the occupied cells of one root whose codes lie in
  `[boundary, next node's boundary)`.  Boundaries are octree-aligned codes.
  A node is named by the largest octree block that starts at its boundary:
  depth `d = D − ⌊tz / n⌋` (0 for boundary 0), where `tz` is the number of
  trailing zero bits of the boundary, and that block's coordinates
  `(c_0, …)` at depth `d` (absolute, so the root is `c >> d`).
* **partition** — the nodes of a root cover its code range without overlap,
  so every occupied cell is in exactly one node.

## Metadata

The array's `zarr.json` carries the extension in two places:

```json
{
  "storage_transformers": [
    {"name": "zarr_octree",
     "configuration": {"version": 2, "max_depth": 7, "min_weight": 4000,
                       "max_weight": 16000, "unit": "vertices"}}
  ],
  "zarr_octree": {"must_understand": true, "version": 2}
}
```

* `storage_transformers` is where Zarr v3 places storage-layer extensions;
  it holds the configuration.
* The top-level `zarr_octree` field with `"must_understand": true` makes a
  reader that does not implement the extension refuse the array.  It is
  needed because zarr-python accepts and ignores storage transformers: without
  it such a reader would open the array and find no chunk at any chunk key,
  i.e. silently read an empty array.

`max_depth` is fixed for the life of the array.  `min_weight`, `max_weight`
and `unit` steer how writers cut runs; readers ignore them.  With unit
`"bytes"` a cell weighs its encoded size plus 16 (its code and offset in the
node index); any other unit is measured by the writer and recorded per node.

Only the `default` chunk key encoding is supported (either separator).

## Objects

All keys are relative to the array.

| key | content |
|-----|---------|
| `zarr.json` | array metadata, as above |
| `o/manifest` | the manifest: every stored node |
| `o/<d>/<c0>/<c1>/…` | node `(d, c)` |

No object exists at any chunk key `c/…`.

### Node object (little endian)

| field | type |
|-------|------|
| magic `ZOCN` | 4 bytes |
| version (2) | uint8 |
| ndim `n` | uint8 |
| depth `d` | uint8 |
| max depth `D` | uint8 |
| cell count `m` | uint32 |
| node coordinates | int64 × n |
| cell codes, sorted | uint64 × m |
| offsets (relative to the payload start) | uint64 × (m + 1) |
| payload | the cells' encoded bytes, concatenated in code order |

A node's bytes depend only on its contents.  A cell with an empty payload
(zero bytes) is present; a cell that is not listed is absent (fill value).

### Manifest object (little endian)

| field | type |
|-------|------|
| magic `ZOCM` | 4 bytes |
| version (2) | uint8 |
| ndim `n` | uint8 |
| max depth `D` | uint8 |
| padding | 1 byte |
| node count `N` | uint64 |
| depth of each node | uint8 × N |
| node coordinates | int64 × N × n |
| code of the node's first occupied cell | uint64 × N |
| code of the node's last occupied cell | uint64 × N |
| node object size in bytes | uint64 × N |
| cells in the node | uint32 × N |
| weight of the node | float64 × N |

## Reading a cell

1. Read the manifest (once per open array).  For each root, sort its nodes
   by boundary (the code where the node's block starts).
2. For cell `x`: root `x >> D`, code `z`.  The node is the last one of that
   root whose boundary is `≤ z`.  If there is none, or `z` lies outside the
   node's first…last occupied codes, the cell is absent — without any storage
   request.
3. Otherwise read the node (whole) and return the bytes listed for `z`, or
   *absent* if `z` is not listed.

A box query is therefore: one manifest read (amortised), then one request per
node the box touches, all independent of each other.  Byte-range requests on
a chunk key (e.g. a `sharding_indexed` codec inside the array) are served
from the cell's bytes.

## Writing

Writers must produce a valid partition; how they choose it is not part of
the format.  The reference writer (`OctreeStore`) buffers chunk writes, then
for each root that received writes either

* rewrites only the nodes whose ranges hold the written cells, if no
  rewritten node of more than one cell exceeds `max_weight`; or
* re-cuts the root: walking its occupied cells in code order, each run ends
  at the boundary that keeps its weight in `[min_weight, max_weight]`, does
  not strand a remainder lighter than `min_weight` where it can avoid it, and
  lies on the coarsest octree boundary available (ties to the heavier run).
  A single cell may exceed `max_weight`; a run is lighter than `min_weight`
  only where its root holds less, or where no cut could satisfy both bounds.

It writes new nodes first, then the manifest, then deletes nodes the
manifest no longer lists.  Edits never merge nodes that shrank; a full
repack restores the partition a fresh write would give.

## Not in this version

* Concurrent writers to one array (the manifest is a single object).
* Atomic multi-object updates: a crash between the node writes and the
  manifest write leaves orphaned or rewritten nodes; a crash after the
  manifest write leaves stale nodes, which readers never reach.
* Chunk key encodings other than `default`.
