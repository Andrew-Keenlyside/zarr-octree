# `zarr_octree` — octree chunk packing for Zarr v3 (draft, format version 1)

A storage-layer extension.  The array's data model is unchanged: shape,
dtype, chunk grid (regular), chunk key encoding, codecs and fill value mean
exactly what they mean without the extension.  Only *where the encoded chunk
bytes live* changes: instead of one storage object per chunk, the chunks are
grouped into octree **nodes**, each holding the encoded bytes of every
occupied chunk in a block of the chunk grid.  Nodes are sized by content, so
a dense region is split into many small blocks and a sparse region is kept
in a few large ones.

## Terms

* **cell** — one chunk of the array, addressed by its chunk coordinates
  `(i, j, k, …)`.  The encoded bytes of a cell are exactly what a regular
  store would hold at the chunk key.
* **max depth `D`** — a depth-0 node (a **root**) covers `2^D` cells along
  every axis; a depth-`D` node covers one cell.  For an `n`-d array a node
  has `2^n` children (an octree for `n = 3`; the format is rank-generic).
* **node `(d, c)`** — the node at depth `d` with node coordinates `c`; it
  covers cells `c · 2^(D−d)` to `(c + 1) · 2^(D−d) − 1` on each axis.  The
  cell `x` lies in node `(d, x >> (D − d))`.
* **partition** — the set of stored nodes.  Every occupied cell is in exactly
  one stored node: no stored node is an ancestor of another (the depth layers
  are mutually exclusive).

## Metadata

The array's `zarr.json` carries the extension in two places:

```json
{
  "storage_transformers": [
    {"name": "zarr_octree",
     "configuration": {"version": 1, "max_depth": 4, "target_bytes": 131072}}
  ],
  "zarr_octree": {"must_understand": true, "version": 1}
}
```

* `storage_transformers` is where Zarr v3 places storage-layer extensions;
  it holds the configuration.
* The top-level `zarr_octree` field with `"must_understand": true` makes a
  reader that does not implement the extension refuse the array.  It is
  needed because zarr-python accepts and ignores storage transformers: without
  it such a reader would open the array and find no chunk at any chunk key,
  i.e. silently read an empty array.

`max_depth` is fixed for the life of the array.  `target_bytes` steers how
writers split nodes; readers ignore it.

Only the `default` chunk key encoding is supported (either separator).

## Objects

All keys are relative to the array.

| key | content |
|-----|---------|
| `zarr.json` | array metadata, as above |
| `o/manifest` | the occupancy manifest: every stored node |
| `o/<d>/<c0>/<c1>/…` | node `(d, c)` |

No object exists at any chunk key `c/…`.

### Node object (little endian)

| field | type |
|-------|------|
| magic `ZOCN` | 4 bytes |
| version (1) | uint8 |
| ndim `n` | uint8 |
| depth `d` | uint8 |
| max depth `D` | uint8 |
| cell count `m` | uint32 |
| node coordinates | int64 × n |
| local cell coordinates, C order | uint32 × m × n |
| offsets (relative to the payload start) | uint64 × (m + 1) |
| payload | the cells' encoded bytes, concatenated in the same order |

Local coordinates are relative to the node's first cell, sorted in C order,
so a node's bytes depend only on its contents, and a reader can find a cell
by binary search on its C-order index within the node's `2^(D−d)`-wide block.  A cell with an empty payload
(zero bytes) is present; a cell that is not listed is absent (fill value).

### Manifest object (little endian)

| field | type |
|-------|------|
| magic `ZOCM` | 4 bytes |
| version (1) | uint8 |
| ndim `n` | uint8 |
| max depth `D` | uint8 |
| padding | 1 byte |
| node count `N` | uint64 |
| depth of each node | uint8 × N |
| node coordinates | int64 × N × n |
| node object size in bytes | uint64 × N |
| cells in the node | uint32 × N |

## Reading a cell

1. Read the manifest (once per open array).
2. For `d = 0 … D`: if node `(d, x >> (D − d))` is in the manifest, read that
   node (whole) and return the bytes listed for local coordinates
   `x − ((x >> (D − d)) << (D − d))`, or *absent* if not listed.
3. If no depth matches, the cell is absent — without any storage request.

A box query is therefore: one manifest read (amortised), then one request per
stored node the box touches, all independent of each other.  Byte-range
requests on a chunk key (e.g. a `sharding_indexed` codec inside the array)
are served from the cell's bytes.

## Writing

Writers must produce a valid partition; how they choose it is not part of
the format.  The reference writer (`OctreeStore`) buffers chunk writes, then
for each root that received writes either

* rewrites only the nodes holding the written cells, if every written cell
  lies in an existing node and no rewritten node exceeds `target_bytes`; or
* rebuilds the root top-down: a node whose cells (payload plus 4·n + 8 bytes
  of index each) exceed `target_bytes` is split into its children, down to
  depth `D`, where a single cell may exceed the target.

It writes new nodes first, then the manifest, then deletes nodes the
manifest no longer lists.  Edits never merge nodes that shrank; a full
repack restores the partition a fresh write would give.

## Not in version 1

* Concurrent writers to one array (the manifest is a single object).
* Atomic multi-object updates: a crash between the node writes and the
  manifest write leaves orphaned or rewritten nodes; a crash after the
  manifest write leaves stale nodes, which readers never reach.
* Chunk key encodings other than `default`.
